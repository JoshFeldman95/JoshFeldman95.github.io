---
layout: post
title:  "Data Science Checklists (Part 3): Design Justice Network Principles and Checklists for Data Scientists"
date:   2020-02-02 14:00:00 -0000
categories: ML
---


We have established that checklists can help data scientists handle the complexity of ensuring their work aligns with a given set of values, but to actually propose a checklist, we need to state explicitly what those values are. Up until now, we have discussed to AI principles in general, without making reference to any specific set of ideals. Our checklist intends to help put the[Design Justice Network](http://designjusticenetwork.org/network-principles) principles into action. In this post we will argue for why the Design Justice Network principles, in particular, may benefit from a checklist and then propose a prototype checklist.

**Note: The following three blog posts will accompany a package called &quot;checklist&quot; that I developed. After installing the package, it will trigger a checklist in the command line whenever users call &quot;git push&quot; if they also insert a short shell script in their shell configuration (available** [**here**](https://github.com/JoshFeldman95/checklist)**).**

**The Complexity of Design Justice**

To properly understand the Design Justice Network Principles, we will first briefly describe design justice. In Sasha Costanza-Chock&#39;s essay &quot;Design Justice: towards an intersectional feminist framework for design theory and practice&quot;, they describe the field as follows:

&quot;Design justice is a field of theory and practice that is concerned with how the design of objects and systems influences the distribution of risks, harms, and benefits among various groups of people. Design justice focuses on the ways that design reproduces, is reproduced by, and/or challenges the matrix of domination (white supremacy, heteropatriarchy, capitalism, and settler colonialism). Design justice is also a growing social movement that aims to ensure a more equitable distribution of design&#39;s benefits and burdens; fair and meaningful participation in design decisions; and recognition of community based design traditions, knowledge, and practices.&quot;

There are a number of key themes to draw out from this definition. First, design justice seeks to understand and limit how the matrix of domination is manifested in design. The matrix of domination is a term[coined by](https://books.google.com/books?id=lbraqrCpA2wC&amp;printsec=frontcover&amp;source=gbs_ge_summary_r&amp;cad=0#v=onepage&amp;q&amp;f=false) Black feminist scholar Patricia Hill Collins to describe the intersecting forms of oppression that individuals experience depending on their various identities. A key feature of describing oppression through the lens of the matrix of domination is that it does not treat intersecting identities as a combination, but as a group in and of itself. The matrix of domination is related to a similar term, intersectionality, which was[proposed](https://chicagounbound.uchicago.edu/cgi/viewcontent.cgi?article=1052&amp;context=uclf) by Black feminist legal scholar Kimberlé Crenshaw. Intersectionality likewise frames oppression and privilege as a series of overlapping identities that exist at once.

Intersectionality and the matrix of domination explicitly frame oppression as a complex phenomenon. It is rare, if not impossible, to find another person who experiences oppression and privilege like oneself. Hence, even if a strategy to fight oppression is effective for one person, there is high uncertainty about how that strategy will apply to someone else. As Costanza-Chock explains:

&quot;Black feminist thought also emphasizes the value of situated knowledge over universalist knowledge. In other words, particular insights about the nature of power, oppression, and resistance come from those who occupy a subjugated standpoint, and knowledge developed from any particular standpoint is always partial knowledge.&quot;

By acknowledging the complexity of the problem directly, this body of Black feminist thought recognizes that a variety of standpoints are always necessary because the knowledge gleaned from one perspective will be inherently limited. This intersectional perspective is central to design justice. To design objects or systems that effectively challenge the matrix of domination, a variety of intersectional voices will be necessary throughout the design process. Instead of outlining a set of instructions to challenge the matrix of domination in all cases, which would be impossible due to the nature of the problem, design justice argues for a procedural approach to these problems. That is not to say design justice is entirely procedural, but rather it recognizes that achieving its substantive goal of more equitable technology can only be accomplished by incorporating the experiences and expertise of marginalized communities. As Costanza-Chock puts it, &quot;we have an ethical imperative to systematically advance the participation of marginalized communities in all stages of the technology design process; through this process, resources and power can be more equitably distributed&quot;.

**Designing the Checklist**

By acknowledging the complexity involved in fighting oppression, design justice is naturally driven towards procedural rather than substantive recommendations, which in turn can be promoted with a checklist. Before attempting to translate the Design Justice Network Principles into a checklist, we first describe some features of a good checklist presented in _The Checklist Manifesto_ (122-124). Here are the key choices and recommendations:

1. **Decide whether the checklist should be a DO-CONFIRM checklist or a READ-DO checklist.** With DO-CONFIRM checklists, team members conduct their work and then, at a natural pause point, they use the checklist to confirm they are following the necessary steps. A READ-DO checklist is used in real time, with the user checking items off as they go.
2. **The checklist should be short.** Ideally somewhere between 5 and 9 items, but this rule can be broken depending on how much time the user has to fill it out. At most, budget 60 to 90 seconds for the checklist to be completed.
3. **Focus on the &quot;the killer items&quot;.** Killer items are steps have the largest negative impact but are still often overlooked.
4. **Use simple, exact, and domain specific language.**
5. **Test the checklist in the real world.**

Our checklist will be a DO-CONFIRM checklist because the items on the checklist will not be completed in real time. Rather, at natural break points in the data scientist&#39;s work, she will complete the checklist, reflecting on whether the project is meeting the Design Justice principles. We make use of the popular version control system, Git, to identify natural pause points. When a data scientist pushes code to a remote Git repository hosting service (i.e. GitHub), the command will trigger a subroutine that requires the user to complete the checklist. For a given project, the checklist will be completed multiple times, which will encourage continuous reflection. With respect to the last recommendation encouraging user testing, since we are only proposing a prototype checklist in this paper, we will not test our checklist in real applications. User testing, however, would be a necessary before releasing this checklist to the public.

To create our checklist, we use the[deon](http://deon.drivendata.org/) checklist as a starting point. The deon checklist divides the data science workflow into five categories: data collection, data storage, analysis, modeling, and deployment (see Iteration 0). Next, we intersect these five categories with the ten Design Justice Network Principles. We also add an additional planning stage because preparation is required to conduct the community collaboration required by the Design Justice principles. In the resulting matrix, we sort the deon checklist items into the appropriate cells, revising them or adding additional checks to better align with the new principles (see Iteration 1). In iteration 2, we display the original deon checklist and our new longer checklist side by side, making changes to better capture the Design Justice principles. Finally, in iteration 3, we compress our longer checklist into as few items as possible. This involves grouping the community collaboration items together and collapsing the privacy items. We also continue to revise the checklist items. After these three iterations, our prototype checklist is as follows:

1. **Missing perspectives:** We have a system to collaborate and build trust with community members, particularly historically marginalized community members, on an ongoing basis.
2. **Diverse Team:** Our team is representative of the community we&#39;re collaborating with and includes historically marginalized voices.
3. **Community Collaboration:** We collaborated with community members on an ongoing basis to:
  1. see what is already working and whether we can help amplify these solutions,
  2. set the objectives for the project,
  3. identify sources of bias that might be introduced during data collection/survey design,
  4. define what successful/beneficial/just outcomes look like and what unsuccessful/harmful/unjust outcomes look like,
  5. select the inputs to our model and define our metrics,
  6. understand what types of explanations will be needed,
  7. identify and prevent unintended uses and abuse of the model,
  8. develop a system to identify if our model inflicts harm, and what should be done if this occurs.
4. **Fair Compensation:** Those who created our data, infrastructure, and hardware were fairly compensated.
5. **Privacy Best Practices** : We proactively considered the privacy of individuals in our training data and of our users (i.e. minimize exposure of personally identifiable information, only collect necessary information, encryption at rest and in transit, data deletion plan, etc.)
6. **Consent:** If we are using data on human subjects, they have provided (a) Freely given, (b) Reversible, (c) Informed, (d) Enthusiastic, and (e) Specific consent.
7. **Met Standards Set by Community:** We have assessed with community members whether our system meets the criteria they defined, disaggregated across intersecting identities (i.e. we meet the criteria not just for Black people and women, but also for Black women)
8. **Honest and Intersectional Representation** : Our visualizations, summary statistics, and reports honestly illustrate outcomes across intersecting identities.
9. **Roll back** : We have tested turning off or rolling back the model in production.
10. **Auditability** : The process of generating the analysis is well documented and reproducible, and we have provided a method for the public sector and civil society to safely access our data and models.
11. **Should This Exist:** We still think we should build this.

While this checklist is still much longer than the 5 to 9 item recommendation provided by Gawande, we believe that the extra length will not pose a problem because data science, unlike surgery and flying an airplane, is not a time sensitive domain. The implemented checklist tool can be found here: [https://github.com/JoshFeldman95/checklist](https://github.com/JoshFeldman95/checklist).

Unfortunately, at this point the checklist itself does not live up to the Design Justice principles and this is an important area for future work. I hope to collaborate with members of the Design Justice community and data scientists from historically marginalized backgrounds to continue iterating on the checklist. Given the level of privilege I bring to this project, there are likely many important issues relating to missed items on the checklist or inaccessible infrastructure that I overlooked. In a sense, it is important for the checklist to pass its own test. Currently, this checklist falls short on items 1, 2, and 3. While it is not there yet, I hope that this checklist eventually helps data scientists better incorporate justice into their work.

**References**

| A. Gawande, The Checklist Manifesto, New York: Metropolitan Books, 2010. |
| --- |
| S. Costanza-Chock, &quot;Design Justice: Towards an Intersectional Feminist Framework for Design Theory and Practice,&quot; in _Proceedings of the Design Research Society_, 2018. |
| S. Glouberman and B. Zimmerman, &quot;Complicated and Complex Systems: What Would Successful Reform of Medicare Look Like?,&quot; Commission on the Future of Health Care in Canada, Canada, 2002. |
| P. Hill Collins, Black Feminist Thought : Knowledge, Consciousness, and the Politics of Empowerment, Boston: Unwin Hyman, 1990. |
| Design Justice Network, &quot;Design Justice Network Principles,&quot; Design Justice Network, [Online]. Available: http://designjusticenetwork.org/network-principles. |
| K. Crenshaw, &quot;Demarginalizing the Intersection of Race and Sex: A Black Feminist Critique of Antidiscrimination Doctrine, Feminist Theory and Antiracist Politics,&quot; _University of Chicago Legal Forum,_ vol. 1989, no. 1, pp. 139-167, 1989. |
